import { Badge, type BadgeProps } from "@/components/ui/badge";
import type { GroupStatus, TaskStatus } from "@/api/jobGroups";

/** Map any job/task status string to a Badge variant (color). */
export function statusVariant(
  status: GroupStatus | TaskStatus | string,
): NonNullable<BadgeProps["variant"]> {
  switch (status) {
    case "done":
      return "success";
    case "running":
      return "default";
    case "queued":
      return "muted";
    case "failed":
    case "dead_letter":
      return "danger";
    default:
      return "outline";
  }
}

/** Human label for a status (dead_letter → "dead letter"). */
export function statusLabel(status: string): string {
  return status === "dead_letter" ? "dead letter" : status;
}

/** A colored status pill. */
export function StatusBadge({
  status,
}: {
  status: GroupStatus | TaskStatus | string;
}) {
  return <Badge variant={statusVariant(status)}>{statusLabel(status)}</Badge>;
}
